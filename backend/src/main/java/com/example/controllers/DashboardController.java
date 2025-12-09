package com.example.controllers;

import com.example.dto.DashboardDto;
import com.example.services.DashboardService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/dashboard")
public class DashboardController {

    @Autowired
    private DashboardService dashboardService;

    @GetMapping(value = "/{studentId}", produces = MediaType.APPLICATION_JSON_VALUE + ";charset=UTF-8")
    public DashboardDto getDashboard(@PathVariable int studentId) {
        return dashboardService.getDashboardForStudent(studentId);
    }
}
