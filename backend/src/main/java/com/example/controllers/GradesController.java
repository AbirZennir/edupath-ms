package com.example.controllers;

import com.example.dto.GradesDto;
import com.example.services.GradesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/grades")
public class GradesController {

    @Autowired
    private GradesService gradesService;

    @GetMapping(value = "/{studentId}", produces = MediaType.APPLICATION_JSON_VALUE + ";charset=UTF-8")
    public GradesDto getGrades(@PathVariable int studentId) {
        return gradesService.getGradesForStudent(studentId);
    }
}
