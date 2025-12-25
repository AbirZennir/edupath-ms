package com.example.services;

import com.example.dto.RecommendationRequest;
import com.example.dto.RecommendationResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;

@Service
public class RecommendationsService {

    @Value("${fastapi.url}")
    private String fastapiUrl;

    private final RestTemplate restTemplate;

    public RecommendationsService() {
        this.restTemplate = new RestTemplate();
    }

    public RecommendationResponse generate(RecommendationRequest req) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<RecommendationRequest> request = new HttpEntity<>(req, headers);

            String url = fastapiUrl + "/ai/recommendations";
            return restTemplate.postForObject(url, request, RecommendationResponse.class);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to fetch recommendations from AI Service: " + e.getMessage());
        }
    }
}
