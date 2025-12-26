package com.example.controllers;

import com.example.dto.AssignmentItemDto;
import com.example.services.AssignmentsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/assignments")
public class AssignmentsController {

    @Autowired
    private AssignmentsService assignmentsService;

    @GetMapping(produces = MediaType.APPLICATION_JSON_VALUE + ";charset=UTF-8")
    public List<AssignmentItemDto> listAssignments(
            @RequestParam(name = "status", required = false) String status,
            @RequestParam(name = "q", required = false) String search) {
        return assignmentsService.listAssignments(status, search);
    }
}
