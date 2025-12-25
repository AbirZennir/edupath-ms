package com.example.entities.learning;

import jakarta.persistence.Embeddable;
import lombok.Data;

import java.io.Serializable;

@Data
@Embeddable
public class StudentVleId implements Serializable {
    private int id_site;
    private int id_student;
    private String code_module;
    private String code_presentation;
}
