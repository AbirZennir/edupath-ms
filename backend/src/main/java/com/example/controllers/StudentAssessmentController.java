package com.example.controllers;

import com.example.entities.learning.StudentAssessment;
import com.example.repositories.StudentAssessmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/student-assessment")
public class StudentAssessmentController {

    @Autowired
    private StudentAssessmentRepository studentAssessmentRepository;

    @GetMapping("/all")
    public List<StudentAssessment> getAllStudentAssessments() {
        return studentAssessmentRepository.findAll();
    }
}

