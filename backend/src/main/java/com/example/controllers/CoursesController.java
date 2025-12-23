package com.example.controllers;

import com.example.dto.CourseItemDto;
import com.example.entities.learning.Courses;
import com.example.services.CoursesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/courses")
public class CoursesController {

    @Autowired
    private CoursesService coursesService;

    @GetMapping("/all")
    public List<Courses> getAllCourses() {
        return coursesService.getAllCourses();
    }

    @GetMapping(produces = MediaType.APPLICATION_JSON_VALUE + ";charset=UTF-8")
    public List<CourseItemDto> listCourses(
            @RequestParam(name = "status", required = false) String status,
            @RequestParam(name = "q", required = false) String search) {
        return coursesService.listCourses(status, search);
    }

    @GetMapping("/student/{studentId}")
    public List<CourseItemDto> getStudentCourses(
            @PathVariable Integer studentId,
            @RequestParam(name = "status", required = false) String status,
            @RequestParam(name = "q", required = false) String search) {
        return coursesService.getStudentCourses(studentId, status, search);
    }
}
