package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ClassDetailDto {
    private String id;
    private String title;
    private String teacher;
    private String semester;
    private int studentsCount;
    private int successRate; // %
    private int activeStudents;
    private int progression; // %
    private int engagement;  // %
}
