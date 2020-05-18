#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;

typedef struct{
    char *type;
    list *options;
}section;

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

int is_network(section *s);
void parse_net_options(list *options, network *net);
void free_section(section *s);
LAYER_TYPE string_to_layer_type(char * type);

#endif
