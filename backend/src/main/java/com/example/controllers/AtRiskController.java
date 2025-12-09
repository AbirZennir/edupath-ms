package com.example.controllers;

import com.example.dto.AtRiskStudentDto;
import com.example.services.AtRiskStudentsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/at-risk")
public class AtRiskController {

    @Autowired
    private AtRiskStudentsService atRiskStudentsService;

    /**
     * Liste des étudiants à risque.
     * Filtre optionnel : niveau = elevé | moyen
     */
    @GetMapping(produces = MediaType.APPLICATION_JSON_VALUE + ";charset=UTF-8")
    public List<AtRiskStudentDto> list(@RequestParam(name = "niveau", required = false) String niveau) {
        return atRiskStudentsService.list(niveau);
    }

    /**
     * Résumé des compteurs par niveau (utile pour badge).
     */
    @GetMapping(value = "/summary", produces = MediaType.APPLICATION_JSON_VALUE + ";charset=UTF-8")
    public Map<String, Long> summary() {
        Map<String, Long> result = new HashMap<>();
        result.put("eleve", atRiskStudentsService.countByLevel("elevé"));
        result.put("moyen", atRiskStudentsService.countByLevel("moyen"));
        result.put("total", (long) atRiskStudentsService.list(null).size());
        return result;
    }
}
