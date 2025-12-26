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
    private double generalProgress; 
    private double deltaWeek;      
    private List<CourseProgressDto> courses;
    private List<AssignmentDto> urgentAssignments;
}
