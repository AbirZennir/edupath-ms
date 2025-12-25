package com.example.dto.ai;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class EvolutionPointDTO {
    private String week;
    private double success;
    private double engagement;
}
