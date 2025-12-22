package com.example.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
public class RecommendationResponse {

    private String studentProfile;

    private String riskLevel;

    private List<RecommendationCategoryDto> categories;
}

