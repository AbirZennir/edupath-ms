package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class DashboardDto {
    private String greetingName;
    private double generalProgress; // 0.0 - 1.0
    private double deltaWeek;       // e.g. +5% -> 0.05
    private List<CourseProgressDto> courses;
    private List<AssignmentDto> urgentAssignments;
}
