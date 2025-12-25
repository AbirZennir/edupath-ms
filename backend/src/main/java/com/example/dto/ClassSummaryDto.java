package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ClassSummaryDto {
    private String id;
    private String title;
    private String semester;
    private String level; // Licence | Master
    private String category; // Informatique, Physique, etc.
    private int studentsCount;
    private int successRate;   // %
    private int completion;    // %
}
