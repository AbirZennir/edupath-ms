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
    private int dueInDays; // ex: 2 => dans 2 jours, -3 => rendu il y a 3 jours
    private String status; // pending | done
}
