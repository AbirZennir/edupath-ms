package com.example.controllers;

import com.example.entities.learning.StudentRegistration;
import com.example.repositories.StudentRegistrationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/student-registration")
public class StudentRegistrationController {

    @Autowired
    private StudentRegistrationRepository studentRegistrationRepository;

    @GetMapping("/all")
    public List<StudentRegistration> getAllStudentRegistrations() {
        return studentRegistrationRepository.findAll();
    }
}

