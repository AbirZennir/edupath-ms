package com.example.entities.learning;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Entity
@Table(name = "studentInfo")
public class StudentInfo {
    @Id
    private int id_student;
    private String code_module;
    private String code_presentation;
    private String gender;
    private String imd_band;
    private String highest_education;
    private String age_band;
    private String num_of_prev_attempts;
    private int studied_credits;
    private String region;
    private String disability;
    private String final_result;
    @OneToMany(mappedBy = "student", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<StudentAssessment> assessments = new ArrayList<>();
}
