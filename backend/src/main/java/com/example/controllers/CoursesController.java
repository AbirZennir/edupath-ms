package com.example.controllers;

import com.example.dto.CourseItemDto;
import com.example.services.CoursesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/courses")
public class CoursesController {

    @Autowired
    private CoursesService coursesService;

    @GetMapping("/all")
    public List<CourseItemDto> getAllCourses() {
        return coursesService.getAllCourses();
    }

    @GetMapping(produces = MediaType.APPLICATION_JSON_VALUE + ";charset=UTF-8")
    public List<CourseItemDto> listCourses(
            @RequestParam(name = "status", required = false) String status,
            @RequestParam(name = "q", required = false) String search
    ) {
        return coursesService.listCourses(status, search);
    }
}
