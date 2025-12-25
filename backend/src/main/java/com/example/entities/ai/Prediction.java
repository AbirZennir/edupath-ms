package com.example.entities.ai;

import com.example.entities.learning.StudentInfo;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "predictions")
@Data
@NoArgsConstructor
@AllArgsConstructor
@IdClass(PredictionId.class)
public class Prediction {

    @Id
    @Column(name = "id_student")
    private Integer idStudent;

    @Id
    @Column(name = "code_presentation")
    private String codePresentation;

    @Column(name = "risk_score")
    private Double riskScore;

    @Column(name = "prediction")
    private Integer prediction; // 0 = success, 1 = at risk

    @Column(name = "cluster")
    private Integer cluster;

    @ManyToOne
    @JoinColumn(name = "id_student", insertable = false, updatable = false)
    private StudentInfo studentInfo;
}