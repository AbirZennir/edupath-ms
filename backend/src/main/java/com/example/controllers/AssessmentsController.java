package com.example.controllers;

import com.example.entities.learning.Assessment;
import com.example.repositories.AssessmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/assessments")
public class AssessmentsController {

    @Autowired
    private AssessmentRepository assessmentRepository;

    @GetMapping("/all")
    public List<Assessment> getAllAssessments() {
        return assessmentRepository.findAll();
    }
}

