package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CourseDTO {
    private String title;
    private String professor;
    private String status; // "En cours" ou "Terminé"
    private String icon; // "math", "science", "computer", etc.
}
