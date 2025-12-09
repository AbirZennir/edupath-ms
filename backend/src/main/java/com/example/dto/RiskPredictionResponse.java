package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class RiskPredictionResponse {
    private int studentId;
    private String codePresentation;
    private double riskScore;
    private int prediction;
    private String modelType;
}
