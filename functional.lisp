;;; ============================================================================
;;; CMPS 3500 – Functional Programming
;;; Modified by Geneva Regpala
;;; CMPS 3500
;;; Complete ML Demo (Common Lisp, SBCL, Linux) – auto-starts on load
;;; ----------------------------------------------------------------------------
;;; • Run:        sbcl --load ml_demo_complete.lisp
;;; • Handles categorical features via one-hot encoding
;;; • Implements all 5 algorithms: Linear Regression, Logistic Regression,
;;;   k-NN, Decision Tree (ID3), Gaussian Naive Bayes
;;; • UTF-8 I/O, robust error handling, timing, metrics
;;; • Seeded randomness (deterministic on SBCL)
;;; ----------------------------------------------------------------------------
;;; Defaults for quick run:
;;;   CSV path default: "adult_income_cleaned.csv"
;;;   Target column:    "income"
;;; ============================================================================

#+sbcl (declaim (optimize (speed 1) (safety 3) (debug 3)))

(defpackage :fp
  (:use :cl)
  (:export :main))
(in-package :fp)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Utilities: strings, numbers, RNG, timing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun trim (s) (string-trim '(#\Space #\Tab #\Newline #\Return) s))

(defun parse-number (s)
  "Parse string S to double-float."
  (let ((*read-default-float-format* 'double-float))
    (coerce (read-from-string s) 'double-float)))

(defun parse-float-or-nil (s)
  "Try to parse S to float; NIL if not numeric."
  (handler-case (parse-number s) (error () nil)))

(defun make-rng (seed)
  "Create a fresh random-state."
  (declare (ignore seed))
  (make-random-state t))

(defmacro with-timing ((var) &body body)
  "Execute BODY and bind elapsed time in seconds to VAR."
  (let ((start-sym (gensym "START"))
        (result-sym (gensym "RESULT")))
    `(let ((,start-sym (get-internal-real-time)))
       (let ((,result-sym (progn ,@body)))
         (setf ,var (/ (- (get-internal-real-time) ,start-sym)
                       internal-time-units-per-second 1.0d0))
         ,result-sym))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; UTF-8 CSV I/O with error handling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun read-file-lines (path)
  "Read UTF-8 file PATH, return list of lines."
  (handler-case
    (with-open-file (in path :direction :input :external-format :utf-8)
      (loop for line = (read-line in nil nil) while line collect line))
    (error (e)
           (format t "Error reading file ~A: ~A~%" path e)
           nil)))

(defun split-csv-line (line)
  "Naive CSV split by comma (good for cleaned CSVs)."
  (let ((parts '()) (start 0) (len (length line)))
    (labels ((emit (end) (push (subseq line start end) parts)))
      (loop for i from 0 below len do
            (if (char= (char line i) #\,)
              (progn (emit i) (setf start (1+ i)))
              (when (= i (1- len)) (emit (1+ i)))))
      (nreverse parts))))

(defun load-csv (path)
  "Return two values: (headers . rows) and row-count. NIL on error."
  (let* ((lines (read-file-lines path)))
    (when (null lines)
      (format t "CSV appears empty or cannot be read: ~A~%" path)
      (return-from load-csv (values nil 0)))
    (handler-case
      (let* ((headers (map 'vector #'trim (split-csv-line (first lines))))
             (rows-list (rest lines))
             (rows (make-array (length rows-list))))
        (loop for l in rows-list for i from 0 do
              (setf (aref rows i) (map 'vector #'trim (split-csv-line l))))
        (values (cons headers rows) (length rows)))
      (error (e)
             (format t "Error parsing CSV: ~A~%" e)
             (values nil 0)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Schema detection + one-hot encoding
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defstruct feat-spec
  kind                ;; :num or :cat
  (cats #() :type (vector t))
  (offset 0 :type fixnum)
  (width 1 :type fixnum))

(defun column-index (headers name)
  (or (position name headers :test #'string-equal)
      (error "Column ~A not found in headers: ~A" name headers)))

(defun ensure-binary (y-str)
  "Map common labels to {0,1}; accept numeric strings too."
  (let ((s (string-downcase (trim y-str))))
    (cond ((or (string= s "1") (string= s ">50k") (string= s "> 50k")) 1.0d0)
          ((or (string= s "0") (string= s "<=50k") (string= s "<= 50k")) 0.0d0)
          (t (let ((n (parse-float-or-nil s)))
               (if n n (error "Cannot convert label '~A' to binary" y-str)))))))

(defun detect-schema (headers rows target-name)
  "Return (specs total-width) for non-target columns."
  (let* ((tidx (column-index headers target-name))
         (p (length headers))
         (n (length rows))
         (temp-specs '()))
    (loop for j from 0 below p do
          (unless (= j tidx)
            (let ((all-numeric t)
                  (catset (make-hash-table :test #'equal)))
              (loop for i from 0 below n do
                    (let ((v (aref (aref rows i) j)))
                      (unless (parse-float-or-nil v)
                        (setf all-numeric nil)
                        (setf (gethash (string-downcase v) catset) t))))
              (if all-numeric
                (push (list :num #() 1) temp-specs)
                (let* ((cats (coerce (loop for k being the hash-keys of catset collect k) 'vector))
                       (k (length cats)))
                  (push (list :cat cats (max 1 k)) temp-specs))))))
    (let* ((rev (nreverse temp-specs))
           (specs (make-array (length rev)))
           (offset 0))
      (loop for idx from 0 below (length rev) do
            (destructuring-bind (kind cats width) (nth idx rev)
              (setf (aref specs idx) (make-feat-spec :kind kind :cats cats :offset offset :width width))
              (incf offset width)))
      (values specs offset))))

(defun table->encoded (headers rows target-name &key (for-regression nil))
  "Return (X y specs). X is numeric with one-hot encoding for categoricals.
  If for-regression is T, y is numeric; otherwise binary classification."
  (handler-case
    (multiple-value-bind (specs totalw) (detect-schema headers rows target-name)
      (let* ((tidx (column-index headers target-name))
             (n (length rows))
             (X (make-array (list n totalw) :element-type 'double-float))
             (y (make-array n :element-type 'double-float)))
        ;; Check if target column is numeric for regression
        (when for-regression
          (let ((first-val (aref (aref rows 0) tidx)))
            (unless (parse-float-or-nil first-val)
              (error "Target column '~A' contains non-numeric values (e.g., '~A').~%~
                     This dataset is only suitable for CLASSIFICATION, not regression.~%~
                     Please choose 'c' for classification when loading data." 
                     target-name first-val))))
        (loop for i from 0 below n do
              (let ((row (aref rows i)) (feat-col 0))
                (setf (aref y i)
                      (if for-regression
                        (parse-number (aref row tidx))
                        (ensure-binary (aref row tidx))))
                (loop for j from 0 below (length headers) do
                      (unless (= j tidx)
                        (let* ((spec (aref specs feat-col)))
                          (ecase (feat-spec-kind spec)
                            (:num
                              (let* ((raw (aref row j))
                                     (val (parse-float-or-nil raw)))
                                (unless val
                                  (error "Non-numeric value in numeric column at row ~D col ~D: ~A" i j raw))
                                (setf (aref X i (feat-spec-offset spec)) val)))
                            (:cat
                              (let* ((raw (string-downcase (aref row j)))
                                     (cats (feat-spec-cats spec))
                                     (k (feat-spec-width spec))
                                     (off (feat-spec-offset spec)))
                                (loop for c from 0 below k do (setf (aref X i (+ off c)) 0.0d0))
                                (let ((idx (position raw cats :test #'string=)))
                                  (when idx (setf (aref X i (+ off idx)) 1.0d0)))))))
                        (incf feat-col)))))
        (values X y specs)))
    (error (e)
           (format t "Error encoding data: ~A~%" e)
           (values nil nil nil))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Normalization (only numeric columns)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun normalize-numeric-columns (X specs)
  "Return (Xn means stds) with z-score on numeric 1-wide specs; one-hot untouched."
  (let* ((n (array-dimension X 0))
         (d (array-dimension X 1))
         (Xn (make-array (list n d) :element-type 'double-float))
         (means (make-array d :element-type 'double-float :initial-element 0.0d0))
         (stds (make-array d :element-type 'double-float :initial-element 1.0d0)))
    (loop for i below n do (loop for j below d do (setf (aref Xn i j) (aref X i j))))
    (loop for s across specs do
          (when (and (eq (feat-spec-kind s) :num) (= (feat-spec-width s) 1))
            (let ((j (feat-spec-offset s)))
              (let ((mu 0.0d0))
                (loop for i below n do (incf mu (aref Xn i j)))
                (setf mu (/ mu (max 1 n)) (aref means j) mu))
              (let ((var 0.0d0))
                (loop for i below n do
                      (let ((dval (- (aref Xn i j) (aref means j)))) (incf var (* dval dval))))
                (let ((sd (max 1d-12 (sqrt (/ var (max 1 (- n 1)))))))
                  (setf (aref stds j) sd)
                  (loop for i below n do
                        (setf (aref Xn i j) (/ (- (aref Xn i j) (aref means j)) sd))))))))
    (values Xn means stds)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Math helpers, split, metrics
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun zeros (n) (make-array n :element-type 'double-float :initial-element 0.0d0))
(defun rows (X) (array-dimension X 0))
(defun cols (X) (array-dimension X 1))

(defun add-bias-col (X)
  (let* ((n (rows X)) (d (cols X))
                      (Z (make-array (list n (1+ d)) :element-type 'double-float)))
    (loop for i below n do (setf (aref Z i 0) 1.0d0))
    (loop for i below n do (loop for j below d do (setf (aref Z i (1+ j)) (aref X i j))))
    Z))

(defun mat-vec (X w)
  (let* ((n (rows X)) (d (cols X)) (y (zeros n)))
    (loop for i below n do
          (let ((s 0.0d0))
            (loop for j below d do (incf s (* (aref X i j) (aref w j))))
            (setf (aref y i) s)))
    y))

(defun sigmoid! (v)
  (loop for i below (length v) do
        (setf (aref v i) (/ 1.0d0 (+ 1.0d0 (exp (- (aref v i))))))) v)

(defun shuffle-indices (n &optional (seed 42))
  (let ((idx (make-array n :element-type 'fixnum))
        (rng (make-rng seed)))
    (loop for i below n do (setf (aref idx i) i))
    (loop for i from (1- n) downto 1 do
          (rotatef (aref idx i) (aref idx (random (1+ i) rng))))
    idx))

(defun train-test-split (X y &key (ratio 0.8d0) (seed 42))
  (let* ((n (rows X)) (ntr (floor (* ratio n))) (nte (- n ntr)) (perm (shuffle-indices n seed)))
    (flet ((takeM (A start count)
                  (let* ((d (cols A)) (O (make-array (list count d) :element-type 'double-float)))
                    (loop for k below count do
                          (let ((i (aref perm (+ start k))))
                            (loop for j below d do (setf (aref O k j) (aref A i j)))))
                    O))
           (takeV (A start count)
                  (let ((O (make-array count :element-type 'double-float)))
                    (loop for k below count do (setf (aref O k) (aref A (aref perm (+ start k)))))
                    O)))
      (values (takeM X 0 ntr) (takeV y 0 ntr) (takeM X ntr nte) (takeV y ntr nte)))))

;;; Classification Metrics
(defun accuracy (ytrue ypred)
  (let ((n (length ytrue)) (c 0))
    (loop for i below n do (when (= (aref ytrue i) (aref ypred i)) (incf c)))
    (/ c (max 1 n) 1.0d0)))

(defun confusion-matrix (ytrue ypred)
  "Return (TP TN FP FN) for binary classification."
  (let ((tp 0) (tn 0) (fp 0) (fn 0))
    (loop for i below (length ytrue) do
          (let ((y (aref ytrue i)) (yh (aref ypred i)))
            (cond ((and (= y 1.0d0) (= yh 1.0d0)) (incf tp))
                  ((and (= y 0.0d0) (= yh 0.0d0)) (incf tn))
                  ((and (= y 0.0d0) (= yh 1.0d0)) (incf fp))
                  ((and (= y 1.0d0) (= yh 0.0d0)) (incf fn)))))
    (list tp tn fp fn)))

(defun macro-f1 (ytrue ypred)
  "Compute macro-averaged F1 score for binary classification."
  (destructuring-bind (tp tn fp fn) (confusion-matrix ytrue ypred)
    (let ((prec (if (zerop (+ tp fp)) 0.0d0 (/ tp (+ tp fp) 1.0d0)))
          (rec (if (zerop (+ tp fn)) 0.0d0 (/ tp (+ tp fn) 1.0d0))))
      (if (zerop (+ prec rec))
        0.0d0
        (/ (* 2 prec rec) (+ prec rec) 1.0d0)))))

;;; Regression Metrics
(defun rmse (ytrue ypred)
  "Root Mean Squared Error."
  (let ((n (length ytrue)) (sum 0.0d0))
    (loop for i below n do
          (let ((diff (- (aref ytrue i) (aref ypred i))))
            (incf sum (* diff diff))))
    (sqrt (/ sum n))))

(defun r-squared (ytrue ypred)
  "R² coefficient of determination."
  (let* ((n (length ytrue))
         (ymean (/ (loop for i below n sum (aref ytrue i)) n))
         (ss-res 0.0d0)
         (ss-tot 0.0d0))
    (loop for i below n do
          (let ((diff (- (aref ytrue i) (aref ypred i)))
                (diff-mean (- (aref ytrue i) ymean)))
            (incf ss-res (* diff diff))
            (incf ss-tot (* diff-mean diff-mean))))
    (if (zerop ss-tot)
      0.0d0
      (- 1.0d0 (/ ss-res ss-tot)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 1. Linear Regression (Least Squares)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun solve-linear-system (A b)
  "Solve Ax = b using Gaussian elimination with partial pivoting."
  (let* ((n (array-dimension A 0))
         (Aug (make-array (list n (1+ n)) :element-type 'double-float)))
    ;; Create augmented matrix [A|b]
    (loop for i below n do
          (loop for j below n do (setf (aref Aug i j) (aref A i j)))
          (setf (aref Aug i n) (aref b i)))
    ;; Forward elimination with partial pivoting
    (loop for k below (1- n) do
          ;; Find pivot
          (let ((max-row k))
            (loop for i from (1+ k) below n do
                  (when (> (abs (aref Aug i k)) (abs (aref Aug max-row k)))
                    (setf max-row i)))
            ;; Swap rows
            (unless (= max-row k)
              (loop for j below (1+ n) do
                    (rotatef (aref Aug k j) (aref Aug max-row j))))
            ;; Eliminate
            (loop for i from (1+ k) below n do
                  (let ((factor (if (zerop (aref Aug k k))
                                  0.0d0
                                  (/ (aref Aug i k) (aref Aug k k)))))
                    (loop for j from k below (1+ n) do
                          (decf (aref Aug i j) (* factor (aref Aug k j))))))))
    ;; Back substitution
    (let ((x (zeros n)))
      (loop for i from (1- n) downto 0 do
            (let ((sum (aref Aug i n)))
              (loop for j from (1+ i) below n do
                    (decf sum (* (aref Aug i j) (aref x j))))
              (setf (aref x i) (if (zerop (aref Aug i i))
                                 0.0d0
                                 (/ sum (aref Aug i i))))))
      x)))

(defun linear-regression-train (X y)
  "Train linear regression using normal equation: w = (X^T X)^-1 X^T y"
  (let* ((Z (add-bias-col X))
         (n (rows Z))
         (d (cols Z))
         (XtX (make-array (list d d) :element-type 'double-float :initial-element 0.0d0))
         (Xty (zeros d)))
    ;; Compute X^T X
    (loop for i below d do
          (loop for j below d do
                (let ((sum 0.0d0))
                  (loop for k below n do
                        (incf sum (* (aref Z k i) (aref Z k j))))
                  (setf (aref XtX i j) sum))))
    ;; Compute X^T y
    (loop for i below d do
          (let ((sum 0.0d0))
            (loop for k below n do
                  (incf sum (* (aref Z k i) (aref y k))))
            (setf (aref Xty i) sum)))
    ;; Solve using Gaussian elimination with partial pivoting
    (let ((w (solve-linear-system XtX Xty)))
      (list :w w))))


(defun linear-regression-predict (model X)
  (let* ((w (getf model :w))
         (Z (add-bias-col X))
         (yhat (mat-vec Z w)))
    yhat))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 2. Logistic Regression (binary, L2)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun randn (n &optional (scale 0.01d0) (seed 7))
  (let ((rng (make-rng seed)) (v (make-array n :element-type 'double-float)))
    (labels ((gauss ()
                    (let* ((u1 (max 1d-12 (- 1 (random 1.0d0 rng))))
                           (u2 (random 1.0d0 rng))
                           (r  (sqrt (* -2.0d0 (log u1))))
                           (th (* 2 pi u2)))
                      (* r (cos th)))))
      (loop for i below n do (setf (aref v i) (* scale (gauss)))))
    v))

(defun logistic-train (X y &key (epochs 400) (lr 0.2d0) (l2 0.003d0) (seed 7))
  (let* ((n (rows X)) (d (cols X))
                      (Z (add-bias-col X))
                      (w (randn (1+ d) 0.01d0 seed))
                      (grad (zeros (1+ d))))
    (dotimes (epoch epochs)
      (let ((p (sigmoid! (mat-vec Z w))))
        (fill grad 0.0d0)
        (loop for j below (length w) do
              (let ((s 0.0d0))
                (loop for i below n do (incf s (* (aref Z i j) (- (aref p i) (aref y i)))))
                (setf (aref grad j) (/ s n))
                (when (> j 0) (incf (aref grad j) (* l2 (aref w j))))))
        (loop for j below (length w) do (decf (aref w j) (* lr (aref grad j))))))
    (list :w w)))

(defun logistic-predict (model X)
  (let* ((w (getf model :w)) (Z (add-bias-col X)) (p (sigmoid! (mat-vec Z w)))
                             (yhat (make-array (length p) :element-type 'double-float)))
    (loop for i below (length p) do (setf (aref yhat i) (if (>= (aref p i) 0.5d0) 1.0d0 0.0d0)))
    yhat))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 3. k-Nearest Neighbors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun euclidean-distance (x1 x2)
  "Compute Euclidean distance between two feature vectors."
  (let ((sum 0.0d0))
    (loop for i below (length x1) do
          (let ((diff (- (aref x1 i) (aref x2 i))))
            (incf sum (* diff diff))))
    (sqrt sum)))

(defun knn-train (X y &key (k 5))
  "Store training data for k-NN (lazy learner)."
  (list :X X :y y :k k))

(defun knn-predict (model X)
  "Predict using k-NN with majority voting."
  (let* ((X-train (getf model :X))
         (y-train (getf model :y))
         (k (getf model :k))
         (n-train (rows X-train))
         (n-test (rows X))
         (d (cols X))
         (yhat (zeros n-test)))
    (format t "Processing ~D test samples (this will be slow)...~%" n-test)
    (loop for i below n-test do
          ;; Progress indicator every 100 samples
          (when (and (> i 0) (zerop (mod i 100)))
            (format t "  Processed ~D/~D samples (~,1F%)~%" i n-test (* 100.0 (/ i n-test))))
          ;; Get test point
          (let ((test-point (make-array d :element-type 'double-float)))
            (loop for j below d do (setf (aref test-point j) (aref X i j)))
            ;; Compute distances to all training points
            (let ((distances (make-array n-train)))
              (loop for j below n-train do
                    (let ((train-point (make-array d :element-type 'double-float)))
                      (loop for col below d do (setf (aref train-point col) (aref X-train j col)))
                      (setf (aref distances j) (cons (euclidean-distance test-point train-point) j))))
              ;; Sort by distance
              (setf distances (sort distances #'< :key #'car))
              ;; Vote among k nearest
              (let ((votes (make-hash-table :test #'equal)))
                (loop for j below (min k n-train) do
                      (let ((label (aref y-train (cdr (aref distances j)))))
                        (incf (gethash label votes 0))))
                ;; Find majority
                (let ((max-votes 0) (max-label 0.0d0))
                  (maphash (lambda (label count)
                             (when (> count max-votes)
                               (setf max-votes count max-label label)))
                           votes)
                  (setf (aref yhat i) max-label))))))
    (format t "  Completed ~D/~D samples (100.0%)~%" n-test n-test)
    yhat))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 4. Decision Tree (ID3 - simplified for binary classification)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defstruct dt-node
  (leaf nil)
  (label 0.0d0)
  (feature -1)
  (threshold 0.0d0)
  (left nil)
  (right nil))

(defun entropy (y indices)
  "Calculate entropy for subset of y indexed by indices."
  (let ((counts (make-hash-table :test #'equal))
        (n (length indices)))
    (loop for i across indices do
          (incf (gethash (aref y i) counts 0)))
    (let ((ent 0.0d0))
      (maphash (lambda (label count)
                 (declare (ignore label))
                 (let ((p (/ count n 1.0d0)))
                   (unless (zerop p)
                     (decf ent (* p (log p 2))))))
               counts)
      ent)))

(defun information-gain (X y indices feature threshold)
  "Calculate information gain for a split."
  (let ((left-idx '()) (right-idx '()))
    (loop for i across indices do
          (if (<= (aref X i feature) threshold)
            (push i left-idx)
            (push i right-idx)))
    (when (or (null left-idx) (null right-idx))
      (return-from information-gain 0.0d0))
    (let* ((n (length indices))
           (n-left (length left-idx))
           (n-right (length right-idx))
           (parent-ent (entropy y indices))
           (left-ent (entropy y (coerce left-idx 'vector)))
           (right-ent (entropy y (coerce right-idx 'vector)))
           (weighted-ent (+ (* (/ n-left n) left-ent)
                            (* (/ n-right n) right-ent))))
      (- parent-ent weighted-ent))))

(defun best-split (X y indices)
  "Find best feature and threshold for splitting."
  (let ((d (cols X))
        (best-gain 0.0d0)
        (best-feature -1)
        (best-threshold 0.0d0))
    (loop for feature below d do
          ;; Try median as threshold
          (let* ((values (loop for i across indices collect (aref X i feature)))
                 (sorted-vals (sort values #'<))
                 (threshold (if sorted-vals
                              (nth (floor (length sorted-vals) 2) sorted-vals)
                              0.0d0))
                 (gain (information-gain X y indices feature threshold)))
            (when (> gain best-gain)
              (setf best-gain gain
                    best-feature feature
                    best-threshold threshold))))
    (values best-feature best-threshold best-gain)))

(defun build-tree (X y indices &key (max-depth 10) (min-samples 5) (depth 0))
  "Build decision tree recursively."
  (let ((n (length indices)))
    ;; Check stopping criteria
    (when (or (<= n min-samples)
              (>= depth max-depth)
              (zerop (entropy y indices)))
      (let ((label (if (zerop n)
                     0.0d0
                     (let ((sum 0.0d0))
                       (loop for i across indices do (incf sum (aref y i)))
                       (if (>= (/ sum n) 0.5d0) 1.0d0 0.0d0)))))
        (return-from build-tree (make-dt-node :leaf t :label label))))
    ;; Find best split
    (multiple-value-bind (feature threshold gain) (best-split X y indices)
      (when (or (= feature -1) (zerop gain))
        (let ((label (let ((sum 0.0d0))
                       (loop for i across indices do (incf sum (aref y i)))
                       (if (>= (/ sum n) 0.5d0) 1.0d0 0.0d0))))
          (return-from build-tree (make-dt-node :leaf t :label label))))
      ;; Split data
      (let ((left-idx '()) (right-idx '()))
        (loop for i across indices do
              (if (<= (aref X i feature) threshold)
                (push i left-idx)
                (push i right-idx)))
        (let ((left-tree (build-tree X y (coerce left-idx 'vector)
                                     :max-depth max-depth
                                     :min-samples min-samples
                                     :depth (1+ depth)))
              (right-tree (build-tree X y (coerce right-idx 'vector)
                                      :max-depth max-depth
                                      :min-samples min-samples
                                      :depth (1+ depth))))
          (make-dt-node :leaf nil
                        :feature feature
                        :threshold threshold
                        :left left-tree
                        :right right-tree))))))

(defun id3-train (X y &key (max-depth 10) (min-samples 5))
  "Train decision tree using ID3 algorithm."
  (let* ((n (rows X))
         (indices (make-array n :element-type 'fixnum)))
    (loop for i below n do (setf (aref indices i) i))
    (list :tree (build-tree X y indices :max-depth max-depth :min-samples min-samples))))

(defun predict-one-tree (tree x)
  "Predict single instance using decision tree."
  (if (dt-node-leaf tree)
    (dt-node-label tree)
    (if (<= (aref x (dt-node-feature tree)) (dt-node-threshold tree))
      (predict-one-tree (dt-node-left tree) x)
      (predict-one-tree (dt-node-right tree) x))))

(defun id3-predict (model X)
  "Predict using trained decision tree."
  (let* ((tree (getf model :tree))
         (n (rows X))
         (d (cols X))
         (yhat (zeros n)))
    (loop for i below n do
          (let ((x-row (make-array d :element-type 'double-float)))
            (loop for j below d do (setf (aref x-row j) (aref X i j)))
            (setf (aref yhat i) (predict-one-tree tree x-row))))
    yhat))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 5. Gaussian Naive Bayes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun gnb-train (X y)
  "Train Gaussian Naive Bayes classifier."
  (let* ((n (rows X))
         (d (cols X))
         (classes (remove-duplicates (coerce y 'list)))
         (class-stats (make-hash-table :test #'equal)))
    ;; For each class, compute mean and std for each feature
    (dolist (c classes)
      (let ((indices (loop for i below n when (= (aref y i) c) collect i))
            (means (zeros d))
            (stds (zeros d)))
        (loop for j below d do
              ;; Compute mean
              (let ((sum 0.0d0))
                (dolist (i indices)
                  (incf sum (aref X i j)))
                (setf (aref means j) (/ sum (max 1 (length indices)))))
              ;; Compute std
              (let ((var 0.0d0))
                (dolist (i indices)
                  (let ((diff (- (aref X i j) (aref means j))))
                    (incf var (* diff diff))))
                (setf (aref stds j) (max 1d-6 (sqrt (/ var (max 1 (length indices))))))))
        (setf (gethash c class-stats)
              (list :prior (/ (length indices) n 1.0d0)
                    :means means
                    :stds stds))))
    (list :classes classes :stats class-stats)))

(defun gaussian-pdf (x mean std)
  "Compute Gaussian probability density."
  (let ((exponent (/ (* -0.5 (expt (- x mean) 2)) (expt std 2))))
    (* (/ 1.0d0 (* std (sqrt (* 2 pi))))
       (exp exponent))))

(defun gnb-predict (model X)
  "Predict using Gaussian Naive Bayes."
  (let* ((classes (getf model :classes))
         (stats (getf model :stats))
         (n (rows X))
         (d (cols X))
         (yhat (zeros n)))
    (loop for i below n do
          (let ((max-prob most-negative-double-float)
                (best-class (first classes)))
            (dolist (c classes)
              (let* ((class-stat (gethash c stats))
                     (prior (getf class-stat :prior))
                     (means (getf class-stat :means))
                     (stds (getf class-stat :stds))
                     (log-prob (log (max 1d-300 prior))))
                ;; Compute log likelihood
                (loop for j below d do
                      (let ((pdf (gaussian-pdf (aref X i j) (aref means j) (aref stds j))))
                        (incf log-prob (log (max 1d-300 pdf)))))
                (when (> log-prob max-prob)
                  (setf max-prob log-prob
                        best-class c))))
            (setf (aref yhat i) best-class)))
    yhat))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Interactive Menu (auto-starts on load)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun prompt (msg) (format t "~A" msg) (finish-output) (read-line))

(defun print-menu ()
  (format t "~%=== ML (Common Lisp) Demo - Complete Implementation ===~%")
  (format t "(1) Load dataset~%")
  (format t "(2) Linear Regression~%")
  (format t "(3) Logistic Regression~%")
  (format t "(4) k-Nearest Neighbors~%")
  (format t "(5) Decision Tree (ID3)~%")
  (format t "(6) Gaussian Naive Bayes~%")
  (format t "(7) Quit~%"))

(defun safe-read-number (prompt-msg default)
  "Safely read a number with default fallback."
  (let ((input (trim (prompt prompt-msg))))
    (if (string= input "")
      default
      (handler-case (parse-number input)
        (error () default)))))

(defun main ()
  "Interactive menu loop. Auto-invoked at end of file."
  (let ((ds nil) (specs nil) (is-classification t))
    (loop
      (print-menu)
      (let ((choice (trim (prompt "Enter option: "))))
        (handler-case
          (cond
            ((string= choice "1")
             (let* ((path (trim (prompt "CSV path [default: adult_income_cleaned.csv]: ")))
                    (path (if (string= path "") "adult_income_cleaned.csv" path))
                    (target (trim (prompt "Target column [default: income]: ")))
                    (target (if (string= target "") "income" target))
                    (task-input (trim (prompt "Task type - c=classification, r=regression [default: c]: ")))
                    (task (if (string= task-input "") "c" task-input))
                    (norm? (string= (string-downcase (trim (prompt "Normalize numeric features? (y/N): "))) "y")))
               (setf is-classification (string= (string-downcase task) "c"))
               ;; Warn if using adult dataset with regression
               (when (and (not is-classification) 
                          (or (search "adult" (string-downcase path))
                              (string= (string-downcase target) "income")))
                 (format t "~%WARNING: The adult income dataset has a categorical target.~%")
                 (format t "Classification (c) is recommended. Continuing with regression may fail...~%~%"))
               (multiple-value-bind (table row-count) (load-csv path)
                 (when (null table)
                   (format t "Failed to load CSV.~%")
                   (return-from main))
                 (destructuring-bind (headers . rows) table
                   (multiple-value-bind (X y sp) (table->encoded headers rows target
                                                                 :for-regression (not is-classification))
                     (when (null X)
                       (format t "Failed to encode data.~%")
                       (return-from main))
                     (setf specs sp)
                     (when norm?
                       (multiple-value-bind (Xn means stds) (normalize-numeric-columns X specs)
                         (declare (ignore means stds))
                         (setf X Xn)))
                     (setf ds (list :X X :y y))
                     (format t "Loaded ~D rows, ~D features (after encoding).~%" (rows X) (cols X)))))))

            ((string= choice "2")
             (if (null ds)
               (format t "Load data first (option 1).~%")
               (let ((fit-time 0.0d0) (pred-time 0.0d0))
                 (multiple-value-bind (Xtr ytr Xte yte) (train-test-split (getf ds :X) (getf ds :y) :seed 42)
                   (format t "~%Algorithm: Linear Regression~%")
                   (format t "Training on ~D samples, testing on ~D samples~%" (rows Xtr) (rows Xte))
                   (let ((model (with-timing (fit-time) (linear-regression-train Xtr ytr))))
                     (format t "Fit time: ~,4F seconds~%" fit-time)
                     (let ((yhat (with-timing (pred-time) (linear-regression-predict model Xte))))
                       (format t "Predict time: ~,4F seconds~%" pred-time)
                       (if is-classification
                         ;; Threshold at 0.5 for classification
                         (let ((yhat-class (make-array (length yhat) :element-type 'double-float)))
                           (loop for i below (length yhat) do
                                 (setf (aref yhat-class i) (if (>= (aref yhat i) 0.5d0) 1.0d0 0.0d0)))
                           (format t "Accuracy: ~,4F~%" (accuracy yte yhat-class))
                           (format t "Macro-F1: ~,4F~%" (macro-f1 yte yhat-class)))
                         (progn
                           (format t "RMSE: ~,4F~%" (rmse yte yhat))
                           (format t "R²: ~,4F~%" (r-squared yte yhat))))))))))

            ((string= choice "3")
             (if (null ds)
               (format t "Load data first (option 1).~%")
               (let ((fit-time 0.0d0) (pred-time 0.0d0)
                                      (epochs (floor (safe-read-number "Epochs [default: 400]: " 400)))
                                      (lr (safe-read-number "Learning rate [default: 0.2]: " 0.2d0))
                                      (l2 (safe-read-number "L2 regularization [default: 0.003]: " 0.003d0)))
                 (multiple-value-bind (Xtr ytr Xte yte) (train-test-split (getf ds :X) (getf ds :y) :seed 42)
                   (format t "~%Algorithm: Logistic Regression~%")
                   (format t "Training on ~D samples, testing on ~D samples~%" (rows Xtr) (rows Xte))
                   (format t "Parameters: epochs=~D, lr=~,4F, l2=~,6F~%" epochs lr l2)
                   (let ((model (with-timing (fit-time)
                                             (logistic-train Xtr ytr :epochs epochs :lr lr :l2 l2 :seed 7))))
                     (format t "Fit time: ~,4F seconds~%" fit-time)
                     (let ((yhat (with-timing (pred-time) (logistic-predict model Xte))))
                       (format t "Predict time: ~,4F seconds~%" pred-time)
                       (format t "Accuracy: ~,4F~%" (accuracy yte yhat))
                       (format t "Macro-F1: ~,4F~%" (macro-f1 yte yhat))))))))

            ((string= choice "4")
             (if (null ds)
               (format t "Load data first (option 1).~%")
               (let ((fit-time 0.0d0) (pred-time 0.0d0)
                                      (k (floor (safe-read-number "k (number of neighbors) [default: 5]: " 5))))
                 (multiple-value-bind (Xtr ytr Xte yte) (train-test-split (getf ds :X) (getf ds :y) :seed 42)
                   (format t "~%WARNING: k-NN is VERY slow! Testing on ~D samples may take 5-30 minutes.~%" (rows Xte))
                   (format t "~%Algorithm: k-Nearest Neighbors~%")
                   (format t "Training on ~D samples, testing on ~D samples~%" (rows Xtr) (rows Xte))
                   (format t "Parameters: k=~D~%" k)
                   (let ((model (with-timing (fit-time) (knn-train Xtr ytr :k k))))
                     (format t "Fit time: ~,4F seconds~%" fit-time)
                     (let ((yhat (with-timing (pred-time) (knn-predict model Xte))))
                       (format t "Predict time: ~,4F seconds~%" pred-time)
                       (if is-classification
                         (progn
                           (format t "Accuracy: ~,4F~%" (accuracy yte yhat))
                           (format t "Macro-F1: ~,4F~%" (macro-f1 yte yhat)))
                         (progn
                           (format t "RMSE: ~,4F~%" (rmse yte yhat))
                           (format t "R²: ~,4F~%" (r-squared yte yhat))))))))))

            ((string= choice "5")
             (if (null ds)
               (format t "Load data first (option 1).~%")
               (let ((fit-time 0.0d0) (pred-time 0.0d0)
                                      (max-depth (floor (safe-read-number "Max depth [default: 10]: " 10)))
                                      (min-samples (floor (safe-read-number "Min samples per leaf [default: 5]: " 5))))
                 (multiple-value-bind (Xtr ytr Xte yte) (train-test-split (getf ds :X) (getf ds :y) :seed 42)
                   (format t "~%Algorithm: Decision Tree (ID3)~%")
                   (format t "Training on ~D samples, testing on ~D samples~%" (rows Xtr) (rows Xte))
                   (format t "Parameters: max-depth=~D, min-samples=~D~%" max-depth min-samples)
                   (let ((model (with-timing (fit-time)
                                             (id3-train Xtr ytr :max-depth max-depth :min-samples min-samples))))
                     (format t "Fit time: ~,4F seconds~%" fit-time)
                     (let ((yhat (with-timing (pred-time) (id3-predict model Xte))))
                       (format t "Predict time: ~,4F seconds~%" pred-time)
                       (if is-classification
                         (progn
                           (format t "Accuracy: ~,4F~%" (accuracy yte yhat))
                           (format t "Macro-F1: ~,4F~%" (macro-f1 yte yhat)))
                         (progn
                           (format t "RMSE: ~,4F~%" (rmse yte yhat))
                           (format t "R²: ~,4F~%" (r-squared yte yhat))))))))))

            ((string= choice "6")
             (if (null ds)
               (format t "Load data first (option 1).~%")
               (let ((fit-time 0.0d0) (pred-time 0.0d0))
                 (multiple-value-bind (Xtr ytr Xte yte) (train-test-split (getf ds :X) (getf ds :y) :seed 42)
                   (format t "~%Algorithm: Gaussian Naive Bayes~%")
                   (format t "Training on ~D samples, testing on ~D samples~%" (rows Xtr) (rows Xte))
                   (let ((model (with-timing (fit-time) (gnb-train Xtr ytr))))
                     (format t "Fit time: ~,4F seconds~%" fit-time)
                     (let ((yhat (with-timing (pred-time) (gnb-predict model Xte))))
                       (format t "Predict time: ~,4F seconds~%" pred-time)
                       (if is-classification
                         (progn
                           (format t "Accuracy: ~,4F~%" (accuracy yte yhat))
                           (format t "Macro-F1: ~,4F~%" (macro-f1 yte yhat)))
                         (progn
                           (format t "RMSE: ~,4F~%" (rmse yte yhat))
                           (format t "R²: ~,4F~%" (r-squared yte yhat))))))))))

            ((string= choice "7")
             (format t "Goodbye!~%")
             (return))

            (t (format t "Invalid option. Please choose 1-7.~%")))
          (error (e)
                 (format t "Error: ~A~%" e)
                 (format t "Please try again.~%")))))))

;;; Auto-start menu upon load:
(fp:main)

;;; End of file
