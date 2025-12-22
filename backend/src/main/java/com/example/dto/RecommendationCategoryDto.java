package com.example.dto;

import lombok.Data;

import java.util.List;

@Data
public class RecommendationCategoryDto {
    private String category;
    private String icon;
    private String color;
    private List<RecommendationItemDto> items;
}

