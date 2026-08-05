package com.techmind.backend.infrastructure.web;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class KeywordControllerTests {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void returnsKeywordsForValidContent() throws Exception {
        mockMvc.perform(post("/api/v1/keywords")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "title": "Introducción a Spring Boot",
                                  "text": "Spring Boot permite crear APIs REST con Java."
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.keywords").isArray())
                .andExpect(jsonPath("$.keywords[0]").value("Spring"));
    }

    @Test
    void rejectsBlankFieldsWithAStableErrorResponse() throws Exception {
        mockMvc.perform(post("/api/v1/keywords")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "title": "",
                                  "text": ""
                                }
                                """))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("VALIDATION_ERROR"))
                .andExpect(jsonPath("$.fieldErrors").isArray());
    }
}
