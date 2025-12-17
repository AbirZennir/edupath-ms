package com.example.controllers;

import com.example.services.CsvImportService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/admin/csv-import")
@CrossOrigin(origins = "*")
public class CsvImportController {

    @Autowired
    private CsvImportService csvImportService;

    @PostMapping("/all")
    public ResponseEntity<?> importAllCsvFiles(@RequestParam String basePath) {
        try {
            csvImportService.importAllCsvFiles(basePath);
            Map<String, String> response = new HashMap<>();
            response.put("message", "Successfully imported all CSV files");
            response.put("status", "success");
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, String> response = new HashMap<>();
            response.put("message", "Failed to import CSV files: " + e.getMessage());
            response.put("status", "error");
            return ResponseEntity.badRequest().body(response);
        }
    }

    @PostMapping("/courses")
    public ResponseEntity<?> importCourses(@RequestParam String filePath) {
        try {
            csvImportService.importCourses(filePath);
            return ResponseEntity.ok(Map.of("message", "Successfully imported courses"));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @PostMapping("/assessments")
    public ResponseEntity<?> importAssessments(@RequestParam String filePath) {
        try {
            csvImportService.importAssessments(filePath);
            return ResponseEntity.ok(Map.of("message", "Successfully imported assessments"));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @PostMapping("/vle")
    public ResponseEntity<?> importVle(@RequestParam String filePath) {
        try {
            csvImportService.importVle(filePath);
            return ResponseEntity.ok(Map.of("message", "Successfully imported VLE"));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @PostMapping("/student-info")
    public ResponseEntity<?> importStudentInfo(@RequestParam String filePath) {
        try {
            csvImportService.importStudentInfo(filePath);
            return ResponseEntity.ok(Map.of("message", "Successfully imported student info"));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @PostMapping("/student-registration")
    public ResponseEntity<?> importStudentRegistration(@RequestParam String filePath) {
        try {
            csvImportService.importStudentRegistration(filePath);
            return ResponseEntity.ok(Map.of("message", "Successfully imported student registration"));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @PostMapping("/student-assessment")
    public ResponseEntity<?> importStudentAssessment(@RequestParam String filePath) {
        try {
            csvImportService.importStudentAssessment(filePath);
            return ResponseEntity.ok(Map.of("message", "Successfully imported student assessment"));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @PostMapping("/student-vle")
    public ResponseEntity<?> importStudentVle(@RequestParam String filePath) {
        try {
            csvImportService.importStudentVle(filePath);
            return ResponseEntity.ok(Map.of("message", "Successfully imported student VLE"));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }
}

