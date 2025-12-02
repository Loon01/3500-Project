package data;

import java.io.*;
import java.util.*;

/**
 * Reads CSV files with headers and handles basic parsing.
 * 
 * Team Members: Solomon Anagha, Geneva Regpala, Adrian Rodriguez, Hermit Singh
 */
public class DataLoader {
    
    private String filePath;
    private List<String> headers;
    private List<String[]> rows;
    
    public DataLoader(String filePath) {
        this.filePath = filePath;
        this.headers = new ArrayList<>();
        this.rows = new ArrayList<>();
    }
    
    /**
     * Load CSV file and return a Dataset
     */
    public Dataset load() throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        
        // Read header line
        String headerLine = reader.readLine();
        if (headerLine == null) {
            reader.close();
            throw new IOException("Empty CSV file");
        }
        
        // Parse headers
        headers = parseCSVLine(headerLine);
        
        // Read data rows
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;  // Skip empty lines
            
            List<String> values = parseCSVLine(line);
            if (values.size() == headers.size()) {
                rows.add(values.toArray(new String[0]));
            }
        }
        
        reader.close();
        
        System.out.println("Loaded " + rows.size() + " rows and " + headers.size() + " columns from " + filePath);
        
        return new Dataset(headers, rows);
    }
    
    /**
     * Parse a single CSV line, handling quoted values
     */
    private List<String> parseCSVLine(String line) {
        List<String> values = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                values.add(current.toString().trim());
                current = new StringBuilder();
            } else {
                current.append(c);
            }
        }
        
        // Add last value
        values.add(current.toString().trim());
        
        return values;
    }
    
    /**
     * Get column names
     */
    public List<String> getHeaders() {
        return headers;
    }
    
    /**
     * Get number of rows loaded
     */
    public int getNumRows() {
        return rows.size();
    }
}
