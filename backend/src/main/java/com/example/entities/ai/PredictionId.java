package com.example.entities.ai;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PredictionId implements Serializable {
    private Integer idStudent;
    private String codePresentation;
}