package com.example.controllers;

import com.example.entities.learning.StudentInfo;
import com.example.repositories.StudentInfoRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/student-info")
public class StudentInfoController {

    @Autowired
    private StudentInfoRepository studentInfoRepository;

    @GetMapping("/all")
    public List<StudentInfo> getAllStudentInfo() {
        return studentInfoRepository.findAll();
    }
}

