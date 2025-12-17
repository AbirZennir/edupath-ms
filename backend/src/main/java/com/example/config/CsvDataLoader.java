package com.example.config;

import com.example.services.CsvImportService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

import java.nio.file.Files;
import java.nio.file.Paths;

@Component
@Order(2) // Run after DataInitializer
public class CsvDataLoader implements CommandLineRunner {
    private static final Logger LOGGER = LoggerFactory.getLogger(CsvDataLoader.class);

    @Autowired
    private CsvImportService csvImportService;

    @Value("${csv.import.enabled:false}")
    private boolean importEnabled;

    @Value("${csv.import.path:./data}")
    private String csvBasePath;

    @Override
    public void run(String... args) {
        if (!importEnabled) {
            LOGGER.info("CSV import is disabled. Set csv.import.enabled=true to enable.");
            return;
        }

        if (!Files.exists(Paths.get(csvBasePath))) {
            LOGGER.warn("CSV import path does not exist: {}", csvBasePath);
            LOGGER.info("Please create the directory and place your CSV files there, or update csv.import.path property");
            return;
        }

        try {
            LOGGER.info("CSV import is enabled. Starting automatic import from: {}", csvBasePath);
            csvImportService.importAllCsvFiles(csvBasePath);
        } catch (Exception e) {
            LOGGER.error("Failed to import CSV files automatically", e);
        }
    }
}

