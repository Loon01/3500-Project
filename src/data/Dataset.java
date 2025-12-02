package data;

import java.util.*;

/**
 * Dataset class - holds tabular data with column names.
 * Similar to a simplified pandas DataFrame.
 * 
 * Team Members: Solomon Anagha, Geneva Regpala, Adrian Rodriguez, Hermit Singh
 */
public class Dataset {
    
    private List<String> columnNames;
    private List<String[]> data;
    
    public Dataset(List<String> columnNames, List<String[]> data) {
        this.columnNames = new ArrayList<>(columnNames);
        this.data = new ArrayList<>();
        for (String[] row : data) {
            this.data.add(row.clone());
        }
    }
    
    /**
     * Get column index by name
     */
    public int getColumnIndex(String columnName) {
        for (int i = 0; i < columnNames.size(); i++) {
            if (columnNames.get(i).equals(columnName)) {
                return i;
            }
        }
        return -1;
    }
    
    /**
     * Get column values as a string array
     */
    public String[] getColumn(String columnName) {
        int colIndex = getColumnIndex(columnName);
        if (colIndex == -1) {
            throw new IllegalArgumentException("Column not found: " + columnName);
        }
        
        String[] column = new String[data.size()];
        for (int i = 0; i < data.size(); i++) {
            column[i] = data.get(i)[colIndex];
        }
        return column;
    }
    
    /**
     * Get column values as doubles (for numeric columns)
     */
    public double[] getColumnAsDouble(String columnName) {
        String[] strValues = getColumn(columnName);
        double[] values = new double[strValues.length];
        
        for (int i = 0; i < strValues.length; i++) {
            try {
                values[i] = Double.parseDouble(strValues[i]);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Cannot convert to double: " + strValues[i] + " in column " + columnName);
            }
        }
        return values;
    }
    
    /**
     * Check if a column contains only numeric values
     */
    public boolean isNumericColumn(String columnName) {
        String[] values = getColumn(columnName);
        for (String val : values) {
            try {
                Double.parseDouble(val);
            } catch (NumberFormatException e) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Get all column names except the specified one(s)
     */
    public List<String> getColumnsExcept(String... excludeColumns) {
        Set<String> excludeSet = new HashSet<>(Arrays.asList(excludeColumns));
        List<String> result = new ArrayList<>();
        
        for (String col : columnNames) {
            if (!excludeSet.contains(col)) {
                result.add(col);
            }
        }
        return result;
    }
    
    /**
     * Extract features (X) and target (y) for modeling
     */
    public DataSplit extractXY(String targetColumn) {
        int targetIndex = getColumnIndex(targetColumn);
        if (targetIndex == -1) {
            throw new IllegalArgumentException("Target column not found: " + targetColumn);
        }
        
        // Get target values
        double[] y = getColumnAsDouble(targetColumn);
        
        // Get feature column names (all except target)
        List<String> featureNames = getColumnsExcept(targetColumn);
        
        // Build X as 2D array
        int numRows = data.size();
        int numFeatures = featureNames.size();
        String[][] XStr = new String[numRows][numFeatures];
        
        for (int i = 0; i < numRows; i++) {
            String[] row = data.get(i);
            int featureIdx = 0;
            for (int j = 0; j < row.length; j++) {
                if (j != targetIndex) {
                    XStr[i][featureIdx++] = row[j];
                }
            }
        }
        
        return new DataSplit(featureNames, XStr, y);
    }
    
    /**
     * Get number of rows
     */
    public int getNumRows() {
        return data.size();
    }
    
    /**
     * Get number of columns
     */
    public int getNumColumns() {
        return columnNames.size();
    }
    
    /**
     * Get column names
     */
    public List<String> getColumnNames() {
        return new ArrayList<>(columnNames);
    }
    
    /**
     * Print dataset info (for debugging)
     */
    public void printInfo() {
        System.out.println("\nDataset Info:");
        System.out.println("  Rows: " + getNumRows());
        System.out.println("  Columns: " + getNumColumns());
        System.out.println("  Column names: " + columnNames);
    }
    
    /**
     * Print first few rows (for debugging)
     */
    public void printHead(int n) {
        System.out.println("\nFirst " + n + " rows:");
        System.out.println(String.join(", ", columnNames));
        for (int i = 0; i < Math.min(n, data.size()); i++) {
            System.out.println(String.join(", ", data.get(i)));
        }
    }
    
    /**
     * Holds X and y data
     */
    public static class DataSplit {
        public List<String> featureNames;
        public String[][] X;
        public double[] y;
        
        public DataSplit(List<String> featureNames, String[][] X, double[] y) {
            this.featureNames = featureNames;
            this.X = X;
            this.y = y;
        }
    }
}
