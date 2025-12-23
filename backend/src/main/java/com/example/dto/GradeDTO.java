package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class GradeDTO {
    private String course;
    private String assessment;
    private double score;
    private double maxScore;
    private String status; // "passed", "failed", "pending"
}
