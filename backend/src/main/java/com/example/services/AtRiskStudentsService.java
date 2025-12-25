package com.example.services;

import com.example.dto.AtRiskStudentDto;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class AtRiskStudentsService {

    private static final Logger log = LoggerFactory.getLogger(AtRiskStudentsService.class);

    private final RestTemplate restTemplate;

    @Value("${fastapi.url}")
    private String fastapiUrl;

    @Autowired
    public AtRiskStudentsService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public List<AtRiskStudentDto> getAtRiskStudents() {
        String url = fastapiUrl + "/ai/at-risk-students";
        try {
            ResponseEntity<String> raw = restTemplate.getForEntity(url, String.class);
            String body = raw.getBody();
            if (body == null || body.isBlank()) {
                log.warn("Received empty response from {}", url);
                return List.of();
            }

            // Log raw response for debugging mapping issues
            log.debug("Raw response from {}: {}", url, body);

            ObjectMapper mapper = new ObjectMapper()
                    .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
                    .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true);

            JsonNode root = mapper.readTree(body);
            List<AtRiskStudentDto> mapped;

            if (root.isArray()) {
                if (root.size() == 0)
                    return List.of();

                // If first element is an object, map directly
                if (root.get(0).isObject()) {
                    mapped = mapper.convertValue(root, new TypeReference<List<AtRiskStudentDto>>() {
                    });
                    return processAndFilter(mapped);
                }

                // If first element is an array (e.g., [[id, nom, classe, ...], ...]) map by
                // indices
                if (root.get(0).isArray()) {
                    log.debug("Response appears to be array-of-arrays; mapping elements by index");
                    List<AtRiskStudentDto> result = new ArrayList<>();
                    for (JsonNode arrNode : root) {
                        try {
                            AtRiskStudentDto dto = new AtRiskStudentDto();
                            if (arrNode.size() > 0 && arrNode.get(0).canConvertToInt())
                                dto.setId(arrNode.get(0).asInt());
                            if (arrNode.size() > 1 && !arrNode.get(1).isNull())
                                dto.setNom(arrNode.get(1).asText());
                            if (arrNode.size() > 2 && !arrNode.get(2).isNull())
                                dto.setClasse(arrNode.get(2).asText());
                            if (arrNode.size() > 3 && !arrNode.get(3).isNull())
                                dto.setModules(arrNode.get(3).asText());
                            if (arrNode.size() > 4 && arrNode.get(4).canConvertToInt())
                                dto.setRisque(arrNode.get(4).asInt());
                            if (arrNode.size() > 5 && !arrNode.get(5).isNull())
                                dto.setDerniereConnexion(arrNode.get(5).asText());
                            if (arrNode.size() > 6 && !arrNode.get(6).isNull())
                                dto.setNiveau(arrNode.get(6).asText());
                            if (arrNode.size() > 7 && !arrNode.get(7).isNull())
                                dto.setAvatar(arrNode.get(7).asText());
                            result.add(dto);
                        } catch (Exception ex) {
                            log.warn("Failed to map array element to AtRiskStudentDto: {}", ex.getMessage());
                        }
                    }
                    mapped = result;
                    return processAndFilter(mapped);
                }

                // Unknown array element type
                log.warn("Unknown array element type returned from {}: first element is {}", url,
                        root.get(0).getNodeType());
                return List.of();
            }

            // Not an array - try to map single object
            if (root.isObject()) {
                AtRiskStudentDto dto = mapper.convertValue(root, AtRiskStudentDto.class);
                mapped = List.of(dto);
                return processAndFilter(mapped);
            }

            log.warn("Unexpected JSON shape from {}", url);
            return List.of();

        } catch (Exception e) {
            log.error("Error fetching or parsing at-risk students from {}: {}", url, e.getMessage());
            return List.of();
        }
    }

    private List<AtRiskStudentDto> processAndFilter(List<AtRiskStudentDto> list) {
        if (list == null || list.isEmpty())
            return List.of();
        List<AtRiskStudentDto> filtered = list.stream()
                .filter(dto -> !isEmptyDto(dto))
                .peek(this::enrichDto)
                .collect(Collectors.toList());
        return filtered.isEmpty() ? List.of() : filtered;
    }

    private void enrichDto(AtRiskStudentDto dto) {
        if (dto.getModules() == null || dto.getModules().isBlank()) {
            // Try to extract module from className (e.g., "BBB 2013B" -> "BBB")
            if (dto.getClasse() != null && !dto.getClasse().isBlank()) {
                String[] parts = dto.getClasse().split("\\s+");
                if (parts.length > 0) {
                    dto.setModules(parts[0]);
                } else {
                    dto.setModules("N/A");
                }
            } else {
                dto.setModules("N/A");
            }
        }
        if (dto.getNiveau() != null) {
            String s = dto.getNiveau().toLowerCase();
            if (s.contains("critical"))
                dto.setNiveau("elevé");
            else if (s.contains("warning"))
                dto.setNiveau("moyen");
            else if (s.contains("safe"))
                dto.setNiveau("faible");
        } else if (dto.getRisque() != null) {
            // Fallback if status is missing but score exists
            if (dto.getRisque() > 70)
                dto.setNiveau("elevé");
            else
                dto.setNiveau("moyen");
        }
    }

    private boolean isEmptyDto(AtRiskStudentDto dto) {
        if (dto == null)
            return true;
        boolean allNull = (dto.getId() == null || dto.getId() == 0)
                && (dto.getNom() == null || dto.getNom().isBlank())
                && (dto.getClasse() == null || dto.getClasse().isBlank())
                && (dto.getModules() == null || dto.getModules().isBlank())
                && (dto.getRisque() == null || dto.getRisque() == 0)
                && (dto.getDerniereConnexion() == null || dto.getDerniereConnexion().isBlank())
                && (dto.getNiveau() == null || dto.getNiveau().isBlank())
                && (dto.getAvatar() == null || dto.getAvatar().isBlank());
        return allNull;
    }
}
