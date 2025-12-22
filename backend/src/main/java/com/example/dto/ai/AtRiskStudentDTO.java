package com.example.dto.ai;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;


@Data
@NoArgsConstructor
@AllArgsConstructor
public class AtRiskStudentDTO {
    private int idStudent;
    private String name;
    private String className;
    private double riskScore;
    private String lastConnection;
    private String status; // "critical" or "attention"
}
