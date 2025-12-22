package com.example.controllers;

import com.example.dto.RecommendationRequest;
import com.example.dto.RecommendationResponse;
import com.example.services.RecommendationsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

@RestController
@RequestMapping("/recommendations")
public class RecommendationsController {

    @Autowired
    private RecommendationsService recommendationsService;

    @PostMapping(produces = MediaType.APPLICATION_JSON_VALUE + ";charset=UTF-8", consumes = MediaType.APPLICATION_JSON_VALUE)
    public RecommendationResponse recommend(@RequestBody RecommendationRequest request) {
        if (request.getRiskScore() == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "risk_score is required");
        }
        return recommendationsService.generate(request);
    }
}
