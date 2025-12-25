package com.example.dto;

import lombok.Data;

@Data
public class RecommendationItemDto {
    private String id;
    private String title;
    private String description;
    private String url;
    private String type;
    private String duration;
    private String difficulty;
    private Integer priority;
}

