package com.example.entities.learning;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
public class Vle {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id_site;
    private String code_module;
    private String code_presentation;
    private String activity_type;
    private int week_from;
    private int week_to;
}
