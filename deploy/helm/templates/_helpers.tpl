{{/* Common name + labels for the release. */}}
{{- define "mangrove.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "mangrove.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "mangrove.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "mangrove.labels" -}}
app.kubernetes.io/name: {{ include "mangrove.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "mangrove.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mangrove.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
