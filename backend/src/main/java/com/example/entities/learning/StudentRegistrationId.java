package com.example.entities.learning;

import jakarta.persistence.Embeddable;
import lombok.Data;

@Data
@Embeddable
public class StudentRegistrationId {
    private String code_module;
    private String code_presentation;
    private int id_student;
}
