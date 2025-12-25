package com.example.controllers;

import com.example.dto.ClassDetailDto;
import com.example.dto.ClassSummaryDto;
import com.example.services.ClassesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/classes")
public class ClassesController {

    @Autowired
    private ClassesService classesService;


    @GetMapping(produces = MediaType.APPLICATION_JSON_VALUE + ";charset=UTF-8")
    public List<ClassSummaryDto> listClasses(
            @RequestParam(name = "level", required = false) String level,
            @RequestParam(name = "category", required = false) String category,
            @RequestParam(name = "q", required = false) String q) {
        return classesService.list(level, category, q);
    }

    @GetMapping(value = "/{id}", produces = MediaType.APPLICATION_JSON_VALUE + ";charset=UTF-8")
    public ClassDetailDto getClassDetail(@PathVariable String id) {
        return classesService.get(id);
    }
}
