package com.example.controllers;

import com.example.dto.AtRiskStudentDto;
import com.example.services.AtRiskStudentsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/at-risk")
public class AtRiskController {

    @Autowired
    private AtRiskStudentsService atRiskStudentsService;

    @GetMapping
    public ResponseEntity<List<AtRiskStudentDto>> getAtRiskStudents() {
        return ResponseEntity.ok(atRiskStudentsService.getAtRiskStudents());
    }
}
