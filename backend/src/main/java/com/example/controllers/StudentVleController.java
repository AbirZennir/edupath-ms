package com.example.controllers;

import com.example.entities.learning.StudentVle;
import com.example.repositories.StudentVleRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/student-vle")
public class StudentVleController {

    @Autowired
    private StudentVleRepository studentVleRepository;

    @GetMapping("/all")
    public List<StudentVle> getAllStudentVle() {
        return studentVleRepository.findAll();
    }
}
