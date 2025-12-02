package com.example.entities.learning;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Entity
@Table(name = "assessments")
public class Assessment {
    @Id
    private int id_assessment;
    private String code_module;
    private String code_presentation;
    private String assessment_type;
    private int date;
    private int weight;

    @ManyToOne
    @JoinColumns({
        @JoinColumn(name = "code_module", referencedColumnName = "code_module", insertable = false, updatable = false),
        @JoinColumn(name = "code_presentation", referencedColumnName = "code_presentation", insertable = false, updatable = false)
    })
    private Courses course;

    @OneToMany(mappedBy = "assessment")
    private List<StudentAssessment> submissions;

}
