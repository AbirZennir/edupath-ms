package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AssignmentItemDto {
    private String title;
    private String course;
    private int dueInDays;  
    private String status;  
}
