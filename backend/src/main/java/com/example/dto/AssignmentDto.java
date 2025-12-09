package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class AssignmentDto {
    private String title;
    private String subtitle;
    private int daysRemaining;
}
