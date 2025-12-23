package com.example.controllers;

import com.example.dto.StudentDashboardDTO;
import com.example.services.StudentDashboardService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/dashboard")
public class StudentDashboardController {

    private final StudentDashboardService dashboardService;

    public StudentDashboardController(StudentDashboardService dashboardService) {
        this.dashboardService = dashboardService;
    }

    @GetMapping("/{studentId}")
    public ResponseEntity<StudentDashboardDTO> getStudentDashboard(@PathVariable Integer studentId) {
        try {
            StudentDashboardDTO dashboard = dashboardService.getDashboard(studentId);
            return ResponseEntity.ok(dashboard);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }
}
