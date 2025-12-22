package com.example.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.Map;

@Data
public class RecommendationRequest {

    @JsonProperty("risk_score")
    private Double riskScore;

    private Integer cluster;

    private Map<String, Object> features;
}

