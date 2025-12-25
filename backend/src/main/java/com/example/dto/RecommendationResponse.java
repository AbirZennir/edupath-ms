package com.example.dto;
import lombok.Data;

import java.util.List;

@Data
public class RecommendationResponse {

    private String studentProfile;

    private String riskLevel;

    private List<RecommendationCategoryDto> categories;
}

