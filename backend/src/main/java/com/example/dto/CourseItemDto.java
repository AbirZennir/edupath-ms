package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CourseItemDto {
    private String id;
    private String codeModule;
    private String codePresentation;
    private String title;
    private int studentsCount;
    private int successRate;
    private int completion;
}
