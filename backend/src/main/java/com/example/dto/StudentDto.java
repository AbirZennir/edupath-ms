package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class StudentDto {
    private int id;
    private String nom; // Mocked name
    private String profil; // Mocked based on result
    private int reussite; // Mocked score
    private int engagement; // Mocked engagement
    private String gender;
    private String region;
    private String finalResult;
    private String avatar; // First letter
}
