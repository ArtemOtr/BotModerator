--
-- PostgreSQL database dump
--

-- Dumped from database version 17.0
-- Dumped by pg_dump version 17.0

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: bans; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bans (
    ban_id bigint NOT NULL,
    user_id bigint NOT NULL,
    reason character varying(64),
    banned_at timestamp without time zone DEFAULT now(),
    chat_id integer
);


ALTER TABLE public.bans OWNER TO postgres;

--
-- Name: bans_ban_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.bans_ban_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.bans_ban_id_seq OWNER TO postgres;

--
-- Name: bans_ban_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.bans_ban_id_seq OWNED BY public.bans.ban_id;


--
-- Name: chats; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.chats (
    chat_id integer NOT NULL,
    tg_chat_id character varying,
    admin_id character varying,
    warn_limit integer,
    spam_mute character varying(40),
    spam_ban boolean DEFAULT false,
    toxic_mute character varying(40),
    toxic_ban boolean DEFAULT false,
    spam_model character varying(20),
    toxic_model character varying(20)
);


ALTER TABLE public.chats OWNER TO postgres;

--
-- Name: chats_chat_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.chats_chat_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.chats_chat_id_seq OWNER TO postgres;

--
-- Name: chats_chat_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.chats_chat_id_seq OWNED BY public.chats.chat_id;


--
-- Name: mutes; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.mutes (
    mute_id bigint NOT NULL,
    user_id bigint NOT NULL,
    reason character varying(64),
    muted_at timestamp without time zone DEFAULT now(),
    mute_dur_sec integer,
    muted_until timestamp without time zone,
    chat_id integer
);


ALTER TABLE public.mutes OWNER TO postgres;

--
-- Name: mutes_mute_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.mutes_mute_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.mutes_mute_id_seq OWNER TO postgres;

--
-- Name: mutes_mute_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.mutes_mute_id_seq OWNED BY public.mutes.mute_id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    user_id bigint NOT NULL,
    username character varying(32) NOT NULL,
    is_banned boolean DEFAULT false,
    is_muted boolean DEFAULT false,
    warn_count smallint DEFAULT 0,
    chat_id integer
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Name: users_user_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_user_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_user_id_seq OWNER TO postgres;

--
-- Name: users_user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_user_id_seq OWNED BY public.users.user_id;


--
-- Name: warns; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.warns (
    warn_id bigint NOT NULL,
    user_id bigint NOT NULL,
    reason character varying(64),
    warned_at timestamp without time zone DEFAULT now(),
    chat_id integer
);


ALTER TABLE public.warns OWNER TO postgres;

--
-- Name: warns_warn_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.warns_warn_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.warns_warn_id_seq OWNER TO postgres;

--
-- Name: warns_warn_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.warns_warn_id_seq OWNED BY public.warns.warn_id;


--
-- Name: bans ban_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bans ALTER COLUMN ban_id SET DEFAULT nextval('public.bans_ban_id_seq'::regclass);


--
-- Name: chats chat_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chats ALTER COLUMN chat_id SET DEFAULT nextval('public.chats_chat_id_seq'::regclass);


--
-- Name: mutes mute_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mutes ALTER COLUMN mute_id SET DEFAULT nextval('public.mutes_mute_id_seq'::regclass);


--
-- Name: users user_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN user_id SET DEFAULT nextval('public.users_user_id_seq'::regclass);


--
-- Name: warns warn_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.warns ALTER COLUMN warn_id SET DEFAULT nextval('public.warns_warn_id_seq'::regclass);


--
-- Name: bans bans_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bans
    ADD CONSTRAINT bans_pkey PRIMARY KEY (ban_id);


--
-- Name: chats chats_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chats
    ADD CONSTRAINT chats_pkey PRIMARY KEY (chat_id);


--
-- Name: mutes mutes_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mutes
    ADD CONSTRAINT mutes_pkey PRIMARY KEY (mute_id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (user_id);


--
-- Name: warns warns_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.warns
    ADD CONSTRAINT warns_pkey PRIMARY KEY (warn_id);


--
-- Name: bans fk_bans_to_chat; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bans
    ADD CONSTRAINT fk_bans_to_chat FOREIGN KEY (chat_id) REFERENCES public.chats(chat_id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: mutes fk_mutes_to_chat; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mutes
    ADD CONSTRAINT fk_mutes_to_chat FOREIGN KEY (chat_id) REFERENCES public.chats(chat_id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: users fk_users_to_chat; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT fk_users_to_chat FOREIGN KEY (chat_id) REFERENCES public.chats(chat_id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: warns fk_warns_to_chat; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.warns
    ADD CONSTRAINT fk_warns_to_chat FOREIGN KEY (chat_id) REFERENCES public.chats(chat_id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: bans user_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bans
    ADD CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES public.users(user_id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: mutes user_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mutes
    ADD CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES public.users(user_id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: warns user_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.warns
    ADD CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES public.users(user_id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- PostgreSQL database dump complete
--

