package com.example.entities.learning;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "studentAssessment")
public class StudentAssessment {
    @EmbeddedId
    private StudentAssessmentId studentAssessmentId;
    private int date_submitted;
    private int is_banked;
    private float score;

    @ManyToOne
    @JoinColumn(name = "id_student", insertable = false, updatable = false)
    private StudentInfo student;

    @ManyToOne
    @JoinColumn(name = "id_assessment", insertable = false, updatable = false)
    private Assessment assessment;

}
