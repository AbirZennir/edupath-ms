package com.example.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.Map;

@Data
public class RecommendationRequest {

    private Double riskScore;

    private String studentId;
    private String codeModule;
    private String codePresentation;

    private Integer cluster;

    private Map<String, Object> features;
}

