package com.example.entities;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "student_risk_prediction")
public class StudentRiskPrediction {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private int studentId;
    private String codePresentation;
    private double riskScore;
    private int prediction;
    private String modelType;
    private Instant createdAt;
}
