package com.example.entities.learning;

import jakarta.persistence.Embeddable;
import lombok.Data;

import java.io.Serializable;

@Data
@Embeddable
public class StudentAssessmentId implements Serializable {
    private int id_student;
    private int id_assessment;
}
